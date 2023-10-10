use ingestion::mongo_utils::MongoDriver;
use dpw
#[tokio::main]
async fn main() -> std::io::Result<()> {
    let mongo: MongoDriver = MongoDriver::new("localhost", 8080, "turbine");
    mongo.connect().await?;

    // create/connect if exists, and flush
    mongo.create_collection("github_data").await?;
    mongo.flush_collection("github_data").await?;

    // set remote urls and download
    let urls = vec!["https://github.com/user/repo/raw/master/file1.zip", "https://github.com/user/repo/raw/master/file2.zip"];
    github_downloader.download_git_zips(urls, "my_repo").await?;


    Ok(())
}