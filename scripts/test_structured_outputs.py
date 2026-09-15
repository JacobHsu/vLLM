from enum import Enum

from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
model = client.models.list().data[0].id


def test_choice():
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}
        ],
        extra_body={"structured_outputs": {"choice": ["positive", "negative"]}},
    )
    print("=== choice ===")
    print(completion.choices[0].message.content)


def test_regex():
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate an example email address for Alan Turing, who works in Enigma. End in .com and new line. Example result: alan.turing@enigma.com\n",
            }
        ],
        extra_body={"structured_outputs": {"regex": r"\w+@\w+\.com\n"}, "stop": ["\n"]},
    )
    print("=== regex ===")
    print(repr(completion.choices[0].message.content))


def test_json():
    class CarType(str, Enum):
        sedan = "sedan"
        suv = "SUV"
        truck = "Truck"
        coupe = "Coupe"

    class CarDescription(BaseModel):
        brand: str
        model: str
        car_type: CarType

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "car-description",
                "schema": CarDescription.model_json_schema(),
            },
        },
    )
    print("=== json ===")
    print(completion.choices[0].message.content)


def test_grammar():
    simplified_sql_grammar = """
        root ::= select_statement

        select_statement ::= "SELECT " column " from " table " where " condition

        column ::= "col_1 " | "col_2 "

        table ::= "table_1 " | "table_2 "

        condition ::= column "= " number

        number ::= "1 " | "2 "
    """
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate an SQL query to show the 'username' and 'email' from the 'users' table.",
            }
        ],
        extra_body={"structured_outputs": {"grammar": simplified_sql_grammar}},
    )
    print("=== grammar ===")
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    test_choice()
    test_regex()
    test_json()
    test_grammar()
