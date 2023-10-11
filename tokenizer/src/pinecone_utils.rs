use pinenut::{Client, Index, models::Vector};
use serde_json::{Value, from_str};


pub struct PinceoneDriver {
    client: Client,
    index: Index,
}

impl PinceoneDriver {
    pub fn new(host: &str, port: u16, db_name: &str) -> Self {
        let client: Client = Client::new(env!("PINECONE_API_KEY"), env!("PINECONE_ENV")).await.unwrap();
        let mut index: Index = client.index(env!("PINECONE_INDEX_NAME"));
        let _ = index.describe().await.unwrap();

        PinceoneDriver { client, index }
    }

    pub async fn insert_data(&self, vec: &Vector) -> Result<(), pinenut::error::Error> {
        /* Sample Vector
        let vec: Vector = Vector{
            id: "B".to_string(), // id given by filename hash
            values: vec![0.5; 32], // embedding vector
            sparse_values: None,
            metadata: None
        };
        */

        match self.index.upsert(String::from("odle"), vec![vec]).await {
            Ok(_) => assert!(true),
            Err(err) => panic!("unable to upsert: {:?}", err)
        }
    }
}
