use ingestion::mongo_utils::MongoDriver;
use mongodb::bson::{doc, Document};
use openai_rust::Client;
use std::error::Error;
use std::env;
use tokio::task;
use tokio::sync::mpsc;
use regex::Regex;


pub struct Tokenizer {
    oai_client: Client,
    mongo_model: &MongoDriver,
    collection: String,
}

impl Tokenizer {

    pub fn new(openai_api_key: &str, mongo_model: &MongoDriver, collection: &str) -> Self {
        OpenAIClient {
            oai_client: Client::new(openai_api_key),
            mongo_model: mongo_model,
            collection: collection.to_string(),
        }
    }

    pub async fn tokenize_collection(&self, collection: &str) -> std::io::Result<()> {
        let result: Vec<Document> = self.mongo_client.get_all_documents(self.collection);
        let col: mongodb::Collection<Document> = self.mongo_model.client.as_ref()
            .unwrap()
            .database(&self.mongo_model.db_name)
            .collection(collection);
    
        let (tx, rx) = mpsc::channel::<Result<Document, String>>(32);
    
        for document in result {
            let tx = tx.clone();
            let col_clone: mongodb::Collection<Document> = col.clone();
            
            task::spawn(async move {
                if let Some(text) = document.get_str("text") {
                    let tokens = tokenize_file(text).await;
                    
                    let update_doc: Document = doc! {
                        "$set": { "tokens": tokens }
                    };
    
                    if let Err(e) = col_clone.update_one(document.clone(), update_doc, None).await {
                        tx.send(Err(format!("Error updating document: {}", e))).await.unwrap();
                    } else {
                        tx.send(Ok(document)).await.unwrap();
                    }
                }
            });
        }
    
        drop(tx);
    
        while let Some(result) = rx.recv().await {
            match result {
                Ok(document) => {
                    print!("Processesed document successfully")
                }
                Err(e) => eprintln!("Error processing document: {}", e),
            }
        }
    
        Ok(())
    }

    async fn tokenize_file(input: &str) -> Vec<Token> {
        let number_regex = Regex::new(r"\b\d+\b").unwrap();
        let identifier_regex = Regex::new(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b").unwrap();
        let operator_regex = Regex::new(r"[\+\-\*/]").unwrap();
    
        let mut tokens: Vec<Token> = Vec::new();
    
        for mat in number_regex.find_iter(input) {
            let token: Token = Token {
                token_type: TokenType::Number,
                value: mat.as_str().to_owned(),
            };
            tokens.push(token);
        }
    
        for mat in identifier_regex.find_iter(input) {
            let token = Token {
                token_type: TokenType::Identifier,
                value: mat.as_str().to_owned(),
            };
            tokens.push(token);
        }
    
        for mat in operator_regex.find_iter(input) {
            let token = Token {
                token_type: TokenType::Operator,
                value: mat.as_str().to_owned(),
            };
            tokens.push(token);
        }
    
        tokens
    }
}

#[derive(Debug)]
enum TokenType {
    Number,
    Identifier,
    Operator,
    // Add more token types as needed
}

#[derive(Debug)]
struct Token {
    token_type: TokenType,
    value: String,
}
