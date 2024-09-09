from openai import OpenAI
import psycopg2
from pgvector.psycopg2 import register_vector
import os
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding


def embed_products(products_json, conn):
    with open(products_json) as products_json:
        products = json.load(products_json)
        products_json.close()

    with conn.cursor() as cur:
        for product in products:
            text_to_embed = f"Product: {product['name']}. Description: {product['description']}. [WARNING] {product['warnings']}"

            embedding = get_embedding(text_to_embed)

            cur.execute(
                """
                INSERT INTO products (name, description, price, category, brand, ingredients, warnings, stock, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    product["name"],
                    product["description"],
                    product["price"],
                    product["category"],
                    product["brand"],
                    product["ingredients"],
                    product["warnings"],
                    product["stock"],
                    embedding,
                ),
            )
        conn.commit()


def query_db(conn, query="fast relief for headaches and muscle pain"):
    query_embedding = get_embedding(query)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT name, description, warnings, embedding <-> %s::vector AS similarity
            FROM products
            ORDER BY similarity
            LIMIT 1;
            """,
            (query_embedding,),
        )

        results = cur.fetchall()
        print()
        for result in results:
            print(
                (
                    f"Name: {result[0]}\n"
                    f"Description: {result[1]}\n"
                    f"Warnings: {result[2]}\n"
                    f"Similarity: {result[3]}\n"
                )
            )


if __name__ == "__main__":
    conn = psycopg2.connect(
        dbname="pgvector_starter",
        user="postgres",
        password="",
        host="localhost",
    )
    register_vector(conn)
    #embed_products("products.json", conn) # if you haven't already
    query_db(conn, "I have a mild headache")

    conn.close()
