//! Starts N Geometry Dash instances, each with its own GDRL_PORT.
//!   launch --exe "C:\...\GeometryDash.exe" --n 4 [--base-port 47000] [--stagger 5]
//!          [--appdata-root C:\gdrl] [--ref C:\path\gdrl_reference.bin]
//! --appdata-root gives every instance its own LOCALAPPDATA (seeded from your real one) so they
//! don't overwrite each other's save files. UNVERIFIED that GD honours LOCALAPPDATA; check that
//! <root>\inst0\GeometryDash gets written to.
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::Duration;

use anyhow::{Context, Result};

fn arg_s(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    arg_s(name).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &to)?;
        } else {
            std::fs::copy(entry.path(), &to)?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let exe: PathBuf = arg_s("--exe").context("--exe <path to GeometryDash.exe> is required")?.into();
    let n: usize = arg("--n", 2);
    let base: u16 = arg("--base-port", gd_protocol::BASE_PORT);
    let stagger: u64 = arg("--stagger", 5);
    let appdata_root = arg_s("--appdata-root").map(PathBuf::from);
    let reference = arg_s("--ref");
    let dir = exe.parent().context("exe has no parent directory")?.to_path_buf();
    let real_local = std::env::var("LOCALAPPDATA").ok().map(PathBuf::from);

    let mut children: Vec<Child> = Vec::new();
    for i in 0..n {
        let port = base + i as u16;
        let mut cmd = Command::new(&exe);
        cmd.current_dir(&dir).env("GDRL_PORT", port.to_string());
        if let Some(r) = &reference {
            cmd.env("GDRL_REF", r);
        }
        if let Some(root) = &appdata_root {
            let la = root.join(format!("inst{i}"));
            let gd = la.join("GeometryDash");
            if !gd.exists() {
                match real_local.as_ref().map(|p| p.join("GeometryDash")) {
                    Some(src) if src.exists() => {
                        println!("seeding {} from {}", gd.display(), src.display());
                        copy_dir(&src, &gd)?;
                    }
                    _ => std::fs::create_dir_all(&gd)?,
                }
            }
            cmd.env("LOCALAPPDATA", &la);
        }
        let child = cmd.spawn().with_context(|| format!("starting {}", exe.display()))?;
        println!("instance {i}: pid {} port {port}", child.id());
        children.push(child);
        std::thread::sleep(Duration::from_secs(stagger));
    }
    println!("All launched. Enter the level in every window. Close this console to stop waiting.");
    for mut c in children {
        let _ = c.wait();
    }
    Ok(())
}
