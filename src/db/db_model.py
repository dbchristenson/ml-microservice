"""
This module defines the database model for the RentApartments table using
SQLAlchemy ORM. It includes the base class and the RentApartments class, which
represents the table structure. The RentApartments class contains various
attributes that correspond to the columns in the table, along with their data
types and constraints.
"""

from sqlalchemy import INTEGER, REAL, VARCHAR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.config.config import settings


class Base(DeclarativeBase):
    pass


class RentApartments(Base):
    """
    RentApartments table model.

    This class defines the structure of the RentApartments table in the
    database. It includes various attributes that correspond to the
    columns in the table, along with their data types and constraints.

    Attributes:
        address (str): Address of the apartment (primary key).
        area (float): Area of the apartment in square meters.
        constraction_year (int): Year of construction.
        rooms (int): Number of rooms in the apartment.
        bedrooms (int): Number of bedrooms in the apartment.
        bathrooms (int): Number of bathrooms in the apartment.
        balcony (str): Balcony information.
        storage (str): Storage information.
        parking (str): Parking information.
        furnished (str): Furnished status.
        garage (str): Garage information.
        garden (str): Garden information.
        energy (str): Energy information.
        facilities (str): Facilities information.
        zip (str): Zip code of the apartment location.
        neighborhood (str): Neighborhood information.
        rent (float): Rent price of the apartment.
    """

    __tablename__ = settings.rent_apartments_table_name

    address: Mapped[str] = mapped_column(VARCHAR(), primary_key=True)
    area: Mapped[float] = mapped_column(REAL())
    constraction_year: Mapped[int] = mapped_column(INTEGER())
    rooms: Mapped[int] = mapped_column(INTEGER())
    bedrooms: Mapped[int] = mapped_column(INTEGER())
    bathrooms: Mapped[int] = mapped_column(INTEGER())
    balcony: Mapped[str] = mapped_column(VARCHAR())
    storage: Mapped[str] = mapped_column(VARCHAR())
    parking: Mapped[str] = mapped_column(VARCHAR())
    furnished: Mapped[str] = mapped_column(VARCHAR())
    garage: Mapped[str] = mapped_column(VARCHAR())
    garden: Mapped[str] = mapped_column(VARCHAR())
    energy: Mapped[str] = mapped_column(VARCHAR())
    facilities: Mapped[str] = mapped_column(VARCHAR())
    zip: Mapped[str] = mapped_column(VARCHAR())
    neighborhood: Mapped[str] = mapped_column(VARCHAR())
    rent: Mapped[float] = mapped_column(REAL())
