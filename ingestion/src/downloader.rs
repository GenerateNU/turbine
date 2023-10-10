use reqwest;
use std::error::Error;
use std::fs::File;
use std::io::copy;
use std::path::Path;
use zip::write::FileOptions;
use zip::ZipWriter;
use mongo_utils::MongoDriver;
use mongodb::bson::{doc, Document};


pub struct GitHubDownloader {
    client: reqwest::Client,
    mongo_model: MongoDriver,
    collection: String,
}

impl GitHubDownloader {
    pub fn new(mongo_model: &MongoDriver, collection: &str) -> Self {
        let client = reqwest::Client::new();
        GitHubDownloader { client, mongo_model, collection }
    }

    pub async fn download_git_zips(
        &self,
        urls: Vec<&str>,
    ) -> Result<(), Box<dyn Error>> {
        for url in urls {
            let response = self.client.get(url).send().await?;

            if response.status() != reqwest::StatusCode::OK {
                eprintln!("Error downloading {}: {:?}", url, response.status());
                continue;
            }
            
            // copy response into memory
            let mut file_buf = Vec::new();
            response.copy_to(&mut file_buf)?;

            // create BSON
            let document = doc! {
                "url": response.url.to_owned(),
                "data": file_buf,
            }

            // insert into Mongostore
            self.mongo_model.insert_document(self.collection, document).await?;

            // clear the buffer for the next iteration
            file_buf.clear();
        }

        Ok(())
    }
}