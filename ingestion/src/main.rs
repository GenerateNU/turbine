use ingestion::mongo_utils::MongoDriver;

async fn main() -> std::io::Result<()> {
    let mongo: MongoDriver = MongoDriver::new("localhost", 8080, "turbine");
    mongo.connect();

    mongo.flush_collection("github_data");

    Ok(())
}