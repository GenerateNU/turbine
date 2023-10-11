use oai::OpenAIClient;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let mongo: MongoDriver = MongoDriver::new("localhost", 8080, "turbine");
    mongo.connect().await?;

    // set collection to tokenize
    let collection = "github_data";
    let oai_key = "generate_2023";
    
    // TODO: create a new collection for each repo, insert documents into sub collections 
    let tokenizer: OpenAIClient = OpenAIClient::new(oai_key, mongo, collection);
    tokenizer.tokenize_collection(collection);
    
    Ok(())
}