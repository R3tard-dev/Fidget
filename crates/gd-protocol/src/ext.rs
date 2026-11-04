//! Step 4 additions to the wire protocol.
use crate::Request;

pub const CMD_INFO: u8 = 4;
pub const ERR_NO_SNAPSHOTS: u16 = 5;

impl Request {
    /// Page `page` of snapshot x positions (OBS_DIM values per page).
    /// Response header: `tick` = total snapshot count, `dx` = spacing between snapshots.
    pub fn info(page: u32) -> Self {
        Self { cmd: CMD_INFO, action: 0, skip: 1, start_x: 0.0, aux: page }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::REQ_LEN;

    #[test]
    fn info_encodes_page() {
        let mut b = [0u8; REQ_LEN];
        Request::info(3).encode(&mut b);
        assert_eq!(b[0], CMD_INFO);
        assert_eq!(u32::from_le_bytes(b[8..12].try_into().unwrap()), 3);
    }
}
