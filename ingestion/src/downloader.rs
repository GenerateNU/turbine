use reqwest;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use mongo_utils::MongoDriver;
use mongodb::bson::doc;
use std::io::Read;
use zip::read::ZipArchive;

use crate::mongo_utils;


pub struct GitHubDownloader {
    client: reqwest::Client,
    mongo_model: &MongoDriver,
    collection: String,
}

impl GitHubDownloader {
    pub fn new(mongo_model: &MongoDriver, collection: &str) -> Self {
        let client = reqwest::Client::new();
        GitHubDownloader { client: client, 
                            mongo_model: mongo_model, 
                            collection: collection.to_string() 
                        }
    }

    pub async fn download_git_zips(
        &self,
        urls: Vec<&str>,
        zip_dirs: Vec<&str>,
    ) -> Result<(), Box<dyn Error>> {
        for (index, url) in urls.iter().enumerate() {
            let response = self.client.get(url.clone()).send().await?;

            if response.status() != reqwest::StatusCode::OK {
                eprintln!("Error downloading {}: {:?}", url, response.status());
                continue;
            }
            let filename: &str = url.split('/').last().unwrap_or("unknown.zip");
            let file_path: std::path::PathBuf = Path::new(zip_dirs[index]).join(filename);
            
            let mut response_body = response.bytes().await?;
            while let Some(chunk) = response_body.concat() {
                let chunk = chunk?;
                file_path.write_all(&chunk)?;
            }

            println!("Downloaded: {}", file_path.display());
        }

        Ok(())
    }

    pub async fn mongo_insert(&self, zip_dir: Vec<&str>, filter_suffix: Vec<&str>) -> mongodb::error::Result<()> {
        let file = File::open(zip_dir)?;
        let reader = std::io::BufReader::new(file);
        let mut archive = ZipArchive::new(reader)?;

        for i in 0..archive.len() {
            let mut file = archive.by_index(i);
            let file_name = file.name().to_string();

            let found_suffix = filter_suffix.iter().any(|&suffix| file_name.ends_with(suffix));
            if !found_suffix {
                continue;
            }

            let mut file_content = Vec::new();
            file.read_to_end(&mut file_content)?;

            let content_str = match std::from_utf8(&file_content) {
                Ok(s) => s.to_string(),
                Err(_) => continue,
            };

            let document = doc! {
                "filename": file_name,
                "text": content_str,
            };
            
            self.mongo_model.insert_document(self.collection.as_str(), document).await?;
        }

        Ok(())
    }
}