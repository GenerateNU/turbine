use ingestion::mongo_utils::MongoDriver;
use mongodb::bson::{doc, Document};
use openai_rust::Client;
use std::error::Error;
use std::env;


pub struct OpenAIClient {
    oai_client: Client,
    mongo_model: &MongoDriver,
    collection: String,
}

impl OpenAIClient {

    pub fn new(openai_api_key: &str, mongo_model: &MongoDriver, collection: &str) -> Self {
        OpenAIClient {
            oai_client: Client::new(openai_api_key),
            mongo_model: mongo_model,
            collection: collection.to_string(),
        }
    }

    pub async fn tokenize_collection(&self, collection: &str) -> std::io::Result<>{
        let result: Vec<Document> = self.mongo_client.get_all_documents(self.collection);
        let col: mongodb::Collection<Document>= self.mongo_model.client.as_ref()
                                                                       .unwrap()
                                                                       .database(&self.mongo_model.db_name)
                                                                       .collection(collection);

        while let Some(result) = cursor.next().await {
            match result {
                Ok(document) => {
                    if let Some(text) = document.get_str("text") {
                        let args: openai_rust::embeddings::EmbeddingsArguments = openai_rust::embeddings::EmbeddingsArguments::new("text-embedding-ada-002", text.to_owned());
                        let embedding: Vec<openai_rust::embeddings::EmbeddingsData> = self.oai_client.create_embeddings(args).await.unwrap().data;
                        
                        let update_doc = doc! {
                            "$set": { "embedding": embedding }
                        };

                        col.update_one(document, update_doc, None).await?;
                    }
                }
                Error(e) => eprintln!("Error processing document: {}", e),
            }
        }
        Ok(())
    }
}
