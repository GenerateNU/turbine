use reqwest;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use mongo_utils::MongoDriver;
use mongodb::bson::doc;
use zip::read::ZipArchive;
use octocrab;
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

            let mut file_content: Vec<_> = Vec::new();
            file.read_to_end(&mut file_content)?;

            let content_str = match std::from_utf8(&file_content) {
                Ok(s) => s.to_string(),
                Err(_) => continue,
            };

            let document: mongodb::bson::Document = doc! {
                "filename": file_name,
                "text": content_str,
            };
            
            self.mongo_model.insert_document(self.collection.as_str(), document).await?;
        }

        Ok(())
    }
    pub async fn download_git_tarballs(
        &self,
        organization: &str,
        token: &str,
        output_dir: &str,
    ) -> Result<(), Box<dyn Error>> {
        let repos = octocrab::instance()
            .repos(organization)
            .list()
            .per_page(100)
            .token(token)
            .send()
            .await?;

        for repo in repos {
            let repo_name = repo.name.unwrap_or_default();
            let tarball_url = format!(
                "https://api.github.com/repos/{}/{}/tarball",
                organization, repo_name
            );

            let response = self.client.get(&tarball_url).header("Authorization", format!("Bearer {}", token)).send().await?;

            if response.status() != reqwest::StatusCode::OK {
                eprintln!("Error downloading {}: {:?}", tarball_url, response.status());
                continue;
            }

            let output_path = Path::new(output_dir).join(format!("{}.tar.gz", repo_name));
            let mut output_file = File::create(&output_path)?;

            let mut response_body = response.bytes().await?;
            while let chunk = response_body.chunk() {
                let chunk = chunk?;
                output_file.write_all(&chunk)?;
            }

            println!("Downloaded: {}", output_path.display());
        }

        Ok(())
    }
}