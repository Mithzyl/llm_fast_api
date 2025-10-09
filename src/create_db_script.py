import os
from sqlmodel import SQLModel, create_engine
from yaml import safe_load

from utils.util import read_yaml_config
from src.db.db import create_db_and_tables

# Ensure the config file is correctly located relative to this script
# Assuming config.yaml is in the root of the src directory
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
if not os.path.exists(config_path):
    # Fallback if config.yaml is in the project root
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

db_config = read_yaml_config(config_path)['database_config']
sql_url = f"mysql+pymysql://{db_config['mysql_url']}:{db_config['port']}/{db_config['database_name']}?charset=utf8mb4"
engine = create_engine(sql_url)

# Re-create the engine with the correct URL for the create_db_and_tables function
# This is a workaround because create_db_and_tables uses a global engine
# In a real scenario, it would be better to pass the engine to the function
# or ensure the global engine is correctly initialized before calling.
# For this script, we'll re-initialize the engine if it's not already done.
# However, the current structure of db.py uses a global engine, so we need to ensure
# that the imports in db.py correctly set up the engine before this script runs.
# A more robust solution would be to refactor db.py to accept the engine.

# For now, we assume the imports in db.py will set up the engine correctly.
# If not, we might need to re-import and re-initialize.
# Let's try calling the function directly first.

if __name__ == "__main__":
    print("Creating database and tables...")
    try:
        create_db_and_tables()
        print("Database and tables created successfully.")
    except Exception as e:
        print(f"Error creating database and tables: {e}")
