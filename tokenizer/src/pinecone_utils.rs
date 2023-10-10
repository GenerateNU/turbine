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

    pub async fn flush_collection(&self, collection_name: &str) -> Result<(), pinenut::error::Error> {
        let collection = self.client.collection(collection_name);
        let result = collection.delete_many(Document::new()).await?;
        println!("Deleted {} documents from {}", result.deleted_count, collection_name);
        Ok(())
    }

    pub async fn remove_collection(&self, collection_name: &str) -> Result<(), pinenut::error::Error> {
        let collection = self.client.database(&self.db_name).collection(collection_name);
        collection.drop(None).await?;
        println!("Dropped {} from {}", collection_name, self.db_name);
        Ok(())
    }

    pub async fn create_collection(&self, collection_name: &str) -> Result<(), pinenut::error::Error> {
        self.client.create_collection(collection_name, self.index).await?;
        println!("Created collection {} in {}", collection_name, self.index);
        Ok(())
    }

    pub async fn collection_size(&self, collection_name: &str) -> Result<i64, pinenut::error::Error> {
        let collection = self.client.database(&self.db_name).collection(collection_name);
        let count = collection.count_documents(Document::new(), None).await?;
        Ok(count)
    }

    pub async fn insert_data(&self, collection_name: &str, json_file: &str, clear: bool) -> Result<(), pinenut::error::Error> {
        let collection = self.client.database(&self.db_name).collection(collection_name);

        if clear {
            self.flush_collection(collection_name).await?;
        }

        let file_contents = std::fs::read_to_string(json_file)?;
        let data: Vec<Value> = from_str(&file_contents)?;

        let mut documents = Vec::new();
        for document in data {
            if let Value::Object(map) = document {
                let bson_doc = Document::try_from(map)?;
                documents.push(bson_doc);
            }
        }

        collection.insert_many(documents, None).await?;
        println!("Inserted {} documents into {} in the {}", documents.len(), collection_name, self.db_name);
        Ok(())
    }

    pub async fn search_query(&self, collection_name: &str, qu: Document, proj: Document, lim: i64, show: bool) -> Result<Vec<Document>, mongodb::error::Error> {
        let collection = self.client.database(&self.db_name).collection(collection_name);
        let cursor = collection.find(qu, None).projection(proj).limit(lim);
        let mut result = Vec::new();

        if show {
            while let Some(doc) = cursor.next().await {
                match doc {
                    Ok(document) => {
                        println!("{:?}", document);
                        result.push(document);
                    }
                    Err(e) => {
                        eprintln!("Error: {:?}", e);
                    }
                }
            }
        } else {
            while let Some(doc) = cursor.next().await {
                match doc {
                    Ok(document) => {
                        result.push(document);
                    }
                    Err(e) => {
                        eprintln!("Error: {:?}", e);
                    }
                }
            }
        }

        Ok(result)
    }
}
