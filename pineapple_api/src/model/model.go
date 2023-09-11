package model

import (
	"github.com/jackc/pgx"
)

type PgModel struct {
	Conn *pgx.Conn
}

type Model interface {
	Book(int64) Book
	AllBooks() ([]Book, error)
	AddBook(Book) (Book, error)
}

func (m *PgModel) Book(id int64) Book {
	book, err := GetBookFromDB(m.Conn, id)

	if err != nil {
		panic(err)
	}

	return book
}

func (m *PgModel) AddBook(book Book) (Book, error) {
	b, err := WriteBookToDb(m.Conn, book)

	if err != nil {
		return Book{}, err
	}

	return b, nil
}

func (m *PgModel) AllBooks() ([]Book, error) {
	books, err := GetAllBooksFromDB(m.Conn)

	if err != nil {
		return []Book{}, err
	}
	return books, nil
}
