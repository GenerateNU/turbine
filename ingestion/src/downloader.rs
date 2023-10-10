use reqwest;
use std::error::Error;
use std::fs::File;
use std::io::copy;
use std::path::Path;
use zip::write::FileOptions;
use zip::ZipWriter;
use mongo_utils::MongoDriver;
use mongodb::bson::{doc, Document};
use std::io::{Read, Write};
use zip::read::ZipArchive;


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
        zip_dirs: Vec<&str>,
    ) -> Result<(), Box<dyn Error>> {
        for (index, url) in urls.iter().enumerate() {
            let response = self.client.get(url).send().await?;

            if response.status() != reqwest::StatusCode::OK {
                eprintln!("Error downloading {}: {:?}", url, response.status());
                continue;
            }
            let filename = url.split('/').last().unwrap_or("unknown_{}.zip", index);
            let file_path = Path::new(zip_dir).join(filename);
            
            let mut response_body = response.bytes_stream();
            while let Some(chunk) = response_body.next().await {
                let chunk = chunk?;
                zip_dir.write_all(&chunk)?;
            }

            println!("Downloaded: {}", file_path.display());
        }

        Ok(())
    }

    pub async fn mongo_insert(&self, zip_dir: Vec<&str>, filter_suffix: Vec<&str>) -> mongodb::error::Result<()> {
        let collection: mongodb::Collection<Document>= self.mongo_model.client.as_ref().unwrap().database(self.mongo_model.db_name).collection(self.collection_name);

        // open the ZIP file
        let file = File::open(zip_dir)?;
        let reader = std::io::BufReader::new(file);
        let mut archive = ZipArchive::new(reader)?;

        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let file_name = file.name().to_string();

            let found_suffix = filter_suffix.iter().any(|&suffix| file_name.ends_with(suffix));
            if !found_suffix {
                continue;
            }

            let mut file_content = Vec::new();
            file.read_to_end(&mut file_content)?;

            let content_str = match str::from_utf8(&file_content) {
                Ok(s) => s.to_string(),
                Err(_) => continue,
            };

            let document = doc! {
                "filename": file_name,
                "text": content_str,
            };
            
            self.mongo_model.insert_document(self.collection, document).await?;
        }

        Ok(())
    }
}