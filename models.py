from sqlalchemy import Column, String, Float, Integer, Date, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()  # Load the .env before reading DATABASE_URL

DATABASE_URL = os.getenv("DATABASE_URL")

Base = declarative_base()

class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    confidence = Column(Float)
    date = Column(Date)
    reason = Column(String)
    tags = Column(String)
    chart_url = Column(String)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
