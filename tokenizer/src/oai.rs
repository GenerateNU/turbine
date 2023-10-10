use ingestion::mongo_utils::MongoDriver;
use mongodb::bson::{doc, Document};
use openai::{Language, OpenaiClient};
use std::error::Error;
use std::env;


pub struct OpenAIClient {
    oai_client: OpenaiClient,
    mongo_model: MongoDriver,
    model: Language,
}

impl OpenAIClient {

    pub fn new(openai_api_key: &str, mongo_model: &MongoDriver) -> Self {
        MongoDriver {
            OpenaiClient::new(openai_api_key),
            mongo_model,
            Language::English,
        }
    }

    pub async fn tokenize_collection(&self, collection: &str) -> std::io::Result<>{
        let result Vec<Document> = self.mongo_client.get_all_documents(self.collection);
        let col: mongodb::Collection<Document>= self.mongo_model.client.as_ref()
                                                                       .unwrap()
                                                                       .database(&self.mongo_model.db_name)
                                                                       .collection(collection);

        while let Some(result) = cursor.next().await {
            match result {
                Ok(document) => {
                    if let Some(text) = document.get_str("text") {

                        let tokens = openai.tokenize(&language_model, text)?;
                        let update_doc = doc! {
                            "$set": { "tokens": tokens }
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
