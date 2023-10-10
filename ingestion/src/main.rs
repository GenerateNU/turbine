use ingestion::mongo_utils::MongoDriver;
use ingestion::downloader::GitHubDownloader;


#[tokio::main]
async fn main() -> std::io::Result<()> {
    let mongo: MongoDriver = MongoDriver::new("localhost", 8080, "turbine");
    mongo.connect().await?;

    // create/connect if exists, and flush
    let collection = "github_data"
    mongo.create_collection(collection).await?;
    mongo.flush_collection(collection).await?;

    // set download links and ingest
    let urls = vec!["https://github.com/user/repo/raw/master/file1.zip", "https://github.com/user/repo/raw/master/file2.zip"];
    let zips = vec!["./repo_1.zip", "./repo_2.zip"];

    let downloader: GithubDownloader = GithubDownloader::new(mongo, collection);
    downloader.download_git_zips(urls, zips).await?;

    let filters = vec![".js", ".go", ".tsx", ".ts", ".yaml", ".yml", ".rs"];
    downloader.mongo_insert(zips, filters).await?;
    
    Ok(())
}