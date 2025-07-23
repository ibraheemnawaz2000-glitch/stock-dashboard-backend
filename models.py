# models.py

from sqlalchemy import Column, String, Float, Integer, Date, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL")  # e.g. from Render Postgres

Base = declarative_base()

class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    confidence = Column(Float)
    date = Column(Date)
    reason = Column(String)
    tags = Column(String)  # Comma-separated string
    chart_url = Column(String)

# DB engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
