from pymongo import MongoClient


class JaxDataloader():
    def __init__(self, mongo_uri, database_name, collection_names, max_length=512, batch_size=4):
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.collection_names = collection_names
        self.max_length = max_length
        self.batch_size = batch_size

        self.client = MongoClient(mongo_uri)
        self.db = self.client[database_name]

    def load_data_from_collection(self, collection_name:str):
        collection = self.db[collection_name]
        cursor = collection.find({})
        data = [document['data'] for document in cursor]
        return data

    def create_batches(self, data):
        batches = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]
        return batches

    def load_and_process_data(self, collection_name):
        data = self.load_data_from_collection(collection_name)
        batches = self.create_batches(data)
        return batches

