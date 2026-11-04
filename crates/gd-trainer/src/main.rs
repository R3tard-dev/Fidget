fn main() -> anyhow::Result<()> {
    println!("protocol v{}, base port {}", gd_protocol::PROTOCOL_VERSION, gd_protocol::BASE_PORT);
    gd_ppo::burn_smoke_test();
    Ok(())
}
