pub const PROTOCOL_VERSION: u32 = 1;
pub const BASE_PORT: u16 = 47000;

pub fn port_for(instance: u16) -> u16 {
    BASE_PORT + instance
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ports_are_sequential() {
        assert_eq!(port_for(0), 47000);
        assert_eq!(port_for(3), 47003);
    }
}
