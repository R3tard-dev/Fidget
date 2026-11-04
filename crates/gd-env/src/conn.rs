use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use anyhow::{Context, Result};
use gd_protocol::{decode_response, Header, Request, OBS_DIM, REQ_LEN, RESP_LEN};

/// Blocking, allocation-free request/response connection to one game instance.
pub struct Connection {
    stream: TcpStream,
    req: [u8; REQ_LEN],
    resp: [u8; RESP_LEN],
}

impl Connection {
    pub fn connect(port: u16, io_timeout: Duration) -> Result<Self> {
        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        let stream = TcpStream::connect_timeout(&addr, Duration::from_secs(3))
            .with_context(|| format!("connecting to {addr} (is the game in a level with RL mode on?)"))?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(io_timeout))?;
        stream.set_write_timeout(Some(io_timeout))?;
        Ok(Self { stream, req: [0; REQ_LEN], resp: [0; RESP_LEN] })
    }

    pub fn call(&mut self, req: &Request, obs: &mut [f32; OBS_DIM]) -> Result<Header> {
        req.encode(&mut self.req);
        self.stream.write_all(&self.req).context("sending request")?;
        self.stream.read_exact(&mut self.resp).context("reading response")?;
        Ok(decode_response(&self.resp, obs)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gd_protocol::*;
    use std::net::TcpListener;

    #[test]
    fn loopback_with_mock_server() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let server = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut req = [0u8; REQ_LEN];
            let mut out = [0u8; RESP_LEN];
            let mut obs = [0f32; OBS_DIM];
            for i in 0..3u32 {
                s.read_exact(&mut req).unwrap();
                obs[0] = req[1] as f32; // echo the action
                let h = Header { version: PROTOCOL_VERSION, obs_dim: OBS_DIM as u16, flags: 0, cmd: req[0], err: 0, tick: i, dx: 1.5 };
                encode_response(&h, &obs, &mut out);
                s.write_all(&out).unwrap();
            }
        });

        let mut c = Connection::connect(port, Duration::from_secs(2)).unwrap();
        let mut obs = [0f32; OBS_DIM];
        let h = c.call(&Request::hello(), &mut obs).unwrap();
        assert_eq!(h.tick, 0);
        let h = c.call(&Request::step(true, 1), &mut obs).unwrap();
        assert_eq!((h.tick, obs[0]), (1, 1.0));
        let h = c.call(&Request::step(false, 1), &mut obs).unwrap();
        assert_eq!((h.tick, obs[0], h.dx), (2, 0.0, 1.5));
        server.join().unwrap();
    }
}
