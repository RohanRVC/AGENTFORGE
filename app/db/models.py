from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    type = Column(String, index=True)
    file_path = Column(Text)
    caption = Column(Text, nullable=True)
    transcript = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("Chunk", back_populates="document")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String, primary_key=True, index=True)
    doc_id = Column(String, ForeignKey("documents.id"))
    text = Column(Text)
    chunk_index = Column(Integer)
    vector_id = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")
