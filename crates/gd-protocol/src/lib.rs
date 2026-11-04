//! Wire format shared with the C++ mod (mod/src/protocol.hpp). Little-endian, fixed size.
use std::fmt;

pub const PROTOCOL_VERSION: u16 = 1;
pub const BASE_PORT: u16 = 47000;
pub const MAGIC: u32 = 0x4C52_4447; // bytes "GDRL"

pub const NUM_OBJ: usize = 16;
pub const OBJ_FEATS: usize = 5;
pub const PLAYER_FEATS: usize = 8;
pub const OBS_DIM: usize = PLAYER_FEATS + NUM_OBJ * OBJ_FEATS; // 88

pub const REQ_LEN: usize = 12;
pub const HEADER_LEN: usize = 20;
pub const RESP_LEN: usize = HEADER_LEN + OBS_DIM * 4; // 372

pub const CMD_HELLO: u8 = 0;
pub const CMD_RESET: u8 = 1;
pub const CMD_STEP: u8 = 2;
pub const CMD_CLOSE: u8 = 3;

pub const FLAG_DONE: u8 = 1;
pub const FLAG_SUCCESS: u8 = 2;
pub const FLAG_DEAD: u8 = 4;

/// Instance `i` listens on BASE_PORT + i.
pub fn port_for(instance: u16) -> u16 {
    BASE_PORT + instance
}

pub fn err_name(code: u16) -> &'static str {
    match code {
        0 => "ok",
        1 => "protocol version mismatch",
        2 => "game busy / not inside a level (request timed out)",
        3 => "no player object",
        4 => "bad command",
        _ => "unknown",
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Request {
    pub cmd: u8,
    pub action: u8,
    pub skip: u8,
    pub start_x: f32,
    pub aux: u32,
}

impl Request {
    pub fn hello() -> Self {
        Self { cmd: CMD_HELLO, action: 0, skip: 1, start_x: 0.0, aux: PROTOCOL_VERSION as u32 }
    }
    pub fn reset(start_x: f32) -> Self {
        Self { cmd: CMD_RESET, action: 0, skip: 1, start_x, aux: 0 }
    }
    pub fn step(click: bool, skip: u8) -> Self {
        Self { cmd: CMD_STEP, action: click as u8, skip, start_x: 0.0, aux: 0 }
    }
    pub fn close() -> Self {
        Self { cmd: CMD_CLOSE, action: 0, skip: 1, start_x: 0.0, aux: 0 }
    }
    pub fn encode(&self, out: &mut [u8; REQ_LEN]) {
        out[0] = self.cmd;
        out[1] = self.action;
        out[2] = self.skip;
        out[3] = 0;
        out[4..8].copy_from_slice(&self.start_x.to_le_bytes());
        out[8..12].copy_from_slice(&self.aux.to_le_bytes());
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Header {
    pub version: u16,
    pub obs_dim: u16,
    pub flags: u8,
    pub cmd: u8,
    pub err: u16,
    pub tick: u32,
    pub dx: f32,
}

impl Header {
    pub fn done(&self) -> bool { self.flags & FLAG_DONE != 0 }
    pub fn success(&self) -> bool { self.flags & FLAG_SUCCESS != 0 }
    pub fn dead(&self) -> bool { self.flags & FLAG_DEAD != 0 }
}

#[derive(Debug)]
pub enum ProtocolError {
    BadMagic(u32),
    VersionMismatch { got: u16, want: u16 },
    ObsDimMismatch { got: u16, want: u16 },
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic(m) => write!(f, "bad magic 0x{m:08X}"),
            Self::VersionMismatch { got, want } => write!(f, "protocol version {got}, expected {want}"),
            Self::ObsDimMismatch { got, want } => write!(f, "obs_dim {got}, expected {want}"),
        }
    }
}
impl std::error::Error for ProtocolError {}

/// Decodes into a caller-owned observation buffer (no allocation).
pub fn decode_response(
    buf: &[u8; RESP_LEN],
    obs: &mut [f32; OBS_DIM],
) -> Result<Header, ProtocolError> {
    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    if magic != MAGIC {
        return Err(ProtocolError::BadMagic(magic));
    }
    let h = Header {
        version: u16::from_le_bytes(buf[4..6].try_into().unwrap()),
        obs_dim: u16::from_le_bytes(buf[6..8].try_into().unwrap()),
        flags: buf[8],
        cmd: buf[9],
        err: u16::from_le_bytes(buf[10..12].try_into().unwrap()),
        tick: u32::from_le_bytes(buf[12..16].try_into().unwrap()),
        dx: f32::from_le_bytes(buf[16..20].try_into().unwrap()),
    };
    if h.version != PROTOCOL_VERSION {
        return Err(ProtocolError::VersionMismatch { got: h.version, want: PROTOCOL_VERSION });
    }
    if h.obs_dim as usize != OBS_DIM {
        return Err(ProtocolError::ObsDimMismatch { got: h.obs_dim, want: OBS_DIM as u16 });
    }
    for (dst, src) in obs.iter_mut().zip(buf[HEADER_LEN..].chunks_exact(4)) {
        *dst = f32::from_le_bytes(src.try_into().unwrap());
    }
    Ok(h)
}

/// Used by the mock server in tests (the real encoder is in the C++ mod).
pub fn encode_response(h: &Header, obs: &[f32; OBS_DIM], out: &mut [u8; RESP_LEN]) {
    out[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    out[4..6].copy_from_slice(&h.version.to_le_bytes());
    out[6..8].copy_from_slice(&h.obs_dim.to_le_bytes());
    out[8] = h.flags;
    out[9] = h.cmd;
    out[10..12].copy_from_slice(&h.err.to_le_bytes());
    out[12..16].copy_from_slice(&h.tick.to_le_bytes());
    out[16..20].copy_from_slice(&h.dx.to_le_bytes());
    for (dst, v) in out[HEADER_LEN..].chunks_exact_mut(4).zip(obs.iter()) {
        dst.copy_from_slice(&v.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ports_are_sequential() {
        assert_eq!(port_for(0), 47000);
        assert_eq!(port_for(3), 47003);
    }

    #[test]
    fn sizes_match_cpp() {
        assert_eq!(OBS_DIM, 88);
        assert_eq!(RESP_LEN, 372);
    }

    #[test]
    fn response_roundtrip() {
        let mut obs = [0f32; OBS_DIM];
        obs[0] = 123.5;
        obs[OBS_DIM - 1] = -1.0;
        let h = Header { version: PROTOCOL_VERSION, obs_dim: OBS_DIM as u16, flags: FLAG_DONE | FLAG_DEAD, cmd: CMD_STEP, err: 0, tick: 77, dx: 1.298 };
        let mut buf = [0u8; RESP_LEN];
        encode_response(&h, &obs, &mut buf);
        let mut obs2 = [0f32; OBS_DIM];
        let h2 = decode_response(&buf, &mut obs2).unwrap();
        assert_eq!(h, h2);
        assert_eq!(obs, obs2);
        assert!(h2.done() && h2.dead() && !h2.success());
    }

    #[test]
    fn rejects_bad_magic() {
        let buf = [0u8; RESP_LEN];
        let mut obs = [0f32; OBS_DIM];
        assert!(matches!(decode_response(&buf, &mut obs), Err(ProtocolError::BadMagic(0))));
    }
}
