use mongodb::{Client, bson::Document, options::FindOptions};

pub struct MongoDriver {
    client: Option<Client>,
    db_name: String,
    host: String,
    port: u16,
}

impl MongoDriver {
    pub fn new(host: &str, port: u16, db_name: &str) -> Self {
        MongoDriver {
            db_name: db_name.to_string(),
            host: host.to_string(),
            port,
            client: None, // Initialize client as None
        }
    }

    pub async fn connect(&mut self) -> Result<(), mongodb::error::Error> {
        let client = Client::with_uri_str(&format!("mongodb://{}:{}", self.host, self.port)).await?;
        self.client = Some(client);
        Ok(())
    }

    pub async fn disconnect(&mut self) {
        self.client = None;
    }

    pub async fn flush_collection(&mut self, collection_name: &str) -> Result<(), mongodb::error::Error> {
        if self.client.is_none() {
            self.connect().await?;
        }
        let collection: mongodb::Collection<Document>= self.client.as_ref().unwrap().database(&self.db_name).collection(collection_name);
        let result = collection.delete_many(Document::new(), None).await?;
        println!("Deleted {} documents from {}", result.deleted_count, collection_name);
        Ok(())
    }
    

    pub async fn create_collection(&mut self, collection_name: &str) -> Result<(), mongodb::error::Error> {
        if self.client.is_none() {
            self.connect().await?;
        }

        self.client.as_ref().unwrap().database(&self.db_name).create_collection(collection_name, None).await?;
        println!("Created collection {} in {}", collection_name, self.db_name);
        Ok(())
    }

    pub async fn collection_size(&self, collection_name: &str) -> Result<u64, mongodb::error::Error> {
        let collection: mongodb::Collection<Document>= self.client.as_ref().unwrap().database(&self.db_name).collection(collection_name);
        let count = collection.count_documents(Document::new(), None).await?;
        Ok(count)
    }


    pub async fn search_query(&self, collection_name: &str, qu: Document, proj: Document, lim: i64, show: bool) -> Result<Vec<Document>, mongodb::error::Error> {
        let collection: mongodb::Collection<Document>= self.client.as_ref().unwrap().database(&self.db_name).collection(collection_name);
        let mut options: FindOptions = FindOptions::default();
        options.projection = Some(proj);
        options.limit = Some(lim);
        

        let cursor: mongodb::Cursor<Document> = collection.find(qu, options).await?;
        let mut result: Vec<Document> = Vec::new();
        while let Some(doc) = cursor.deserialize_current() {
            match doc {
                Ok(document) => {
                    if show {
                        println!("{:?}", document);
                    }
                    result.push(document);
                }
                Err(e) => {
                    eprintln!("Error: {:?}", e);
                }
            }
        }

    
        Ok(result)
    }
}
