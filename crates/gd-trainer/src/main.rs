fn main() -> anyhow::Result<()> {
    if std::env::args().any(|a| a == "--help" || a == "-h") {
        println!("{}", gd_trainer::train::USAGE);
        return Ok(());
    }
    let args = gd_trainer::train::Args::from_env();
    if args.eval > 0 {
        return gd_trainer::extra::eval(&args);
    }
    let s = gd_trainer::train::run(&args)?;
    println!(
        "done: {} iterations, stage {}/{}, curriculum {}",
        s.iterations, s.frontier + 1, s.stages, if s.finished { "COMPLETE" } else { "not finished" }
    );
    Ok(())
}
