import json
import pandas as pd
from sqlalchemy import create_engine, exc
import hydra
from omegaconf import DictConfig

# Function to read database configuration from JSON file
def get_database_config_from_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
        return data["database"]

# Function to connect to the database
def connect_to_db(config_data, db_name=None):
    # Use the db_name argument if provided, otherwise use the default name from config_data
    database_name = db_name if db_name else config_data['name']
    connection_string = f"mariadb+mariadbconnector://{config_data['user']}:{config_data['password']}@{config_data['host']}:{config_data['port']}/{database_name}"
    engine = create_engine(connection_string)
    return engine

# Function to get all table names in the database
def get_table_names(connection):
    return pd.read_sql_query("SHOW TABLES;", connection)

# Function to perform a read operation with a rollback mechanism
def read_table_data(sql_query, connection):
    try:
        # Start a transaction explicitly
        trans = connection.begin()
        df = pd.read_sql_query(sql_query, connection)
        # Commit the transaction after a successful read
        trans.commit()
        return df
    except exc.SQLAlchemyError as e:
        # If a SQLAlchemy error occurs, roll back the transaction
        trans.rollback()
        print(f"An SQLAlchemy error occurred: {e}")
        return None
    except Exception as e:
        # For any other exception, roll back the transaction
        trans.rollback()
        print(f"An unexpected error occurred: {e}")
        return None

# Hydra main function for running the script
@hydra.main(config_path="E:\Code\eptic\config", config_name="main", version_base=None)
def main(config: DictConfig):
    config_data = get_database_config_from_json(config.db.credentials)

    print(f"Connecting to database: {config_data['name']}")

    # Connect to the database using the engine
    engine = connect_to_db(config_data)
    with engine.connect() as conn:
        # Get a list of all table names
        table_names_df = get_table_names(conn)
        table_names = table_names_df[table_names_df.columns[0]].tolist()  # Adjust the index if necessary

        # Define SQL query to select all rows from the 'events' table
        sql_query = "SELECT * FROM events"

        # Perform the read operation
        df = read_table_data(sql_query, conn)
        if df is not None:
            # Specify the output file path for the events data
            output_path = "events_data.xlsx"
            # Export the DataFrame to an Excel file
            df.to_excel(output_path, index=False)

# Entry point of the script
if __name__ == "__main__":
    main()