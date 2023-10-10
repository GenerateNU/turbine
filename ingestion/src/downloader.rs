use reqwest;
use std::error::Error;
use std::fs::File;
use std::io::copy;
use std::path::Path;
use zip::write::FileOptions;
use zip::ZipWriter;

pub struct GitHubDownloader {
    client: reqwest::Client,
}

impl GitHubDownloader {
    pub fn new() -> Self {
        let client = reqwest::Client::new();
        GitHubDownloader { client }
    }

    pub async fn download_and_zip_urls(
        &self,
        urls: Vec<(&str, &str)>,
        zip_file_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        let file = File::create(zip_file_path)?;
        let mut zip = ZipWriter::new(file);
        let options = FileOptions::default()
            .unix_permissions(0o755)
            .compression_method(zip::CompressionMethod::Stored);

        for (url, file_name) in urls {
            let response = self.client.get(url).send().await?;

            if response.status() != reqwest::StatusCode::OK {
                eprintln!("Error downloading {}: {:?}", url, response.status());
                continue;
            }

            let mut file_buf = Vec::new();
            response.copy_to(&mut file_buf)?;

            zip.start_file(file_name, options)?;
            zip.write_all(&file_buf)?;
        }

        Ok(())
    }
}