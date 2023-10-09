use redis::{Client, Commands};
use std::thread;

pub struct RedisDriver {
    client: redis::Client,
}

impl RedisDriver {
    pub fn new(host: &str, port: u16) -> Self {
        let url = format!("redis://{}:{}", host, port);
        let client = Client::open(url).expect("Failed to create Redis client");
        RedisDriver { client }
    }

    pub fn set(&self, key: &str, value: &str) -> redis::RedisResult<()> {
        let mut conn = self.client.get_connection()?;
        conn.set(key, value)?;
        Ok(())
    }

    pub fn get(&self, key: &str) -> redis::RedisResult<Option<String>> {
        let mut conn = self.client.get_connection()?;
        let result: Option<String> = conn.get(key)?;
        Ok(result)
    }

    pub fn delete(&self, key: &str) -> redis::RedisResult<()> {
        let mut conn = self.client.get_connection()?;
        conn.del(key)?;
        Ok(())
    }

    pub fn run_queries_in_parallel(&self, keys: Vec<String>) -> Vec<redis::RedisResult<Option<String>>> {
        let mut handles = vec![];

        for key in keys {
            let client = self.client.clone();
            let handle = thread::spawn(move || {
                let conn = client.get_connection();
                conn.and_then(|mut conn| conn.get(&key))
            });

            handles.push(handle);
        }

        let mut results = vec![];
        for handle in handles {
            results.push(handle.join().unwrap());
        }

        results
    }
}
