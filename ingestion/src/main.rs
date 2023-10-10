use ingestion::mongo_utils::MongoDriver;

async fn main() -> std::io::Result<()> {
    let mongo: MongoDriver = MongoDriver::new("localhost", 8080, "turbine");
    mongo.connect();

    // create/connect if exists, and flush
    mongo.create_collection("github_data");
    mongo.flush_collection("github_data");

    Ok(())
}