import chromadb

client = chromadb.HttpClient(host="localhost", port=8001)

# Crear colección de prueba
collection = client.get_or_create_collection("test_memories")

# Insertar una memoria de prueba
collection.add(
    documents=["El usuario prefiere trabajar en bloques de 90 minutos"],
    metadatas=[{"type": "preference", "session": 1}],
    ids=["mem_001"]
)

# Consultar
results = collection.query(
    query_texts=["bloques de tiempo productividad"],
    n_results=1
)

print("ChromaDB funcionando!")
print(results["documents"])