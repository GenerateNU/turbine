package model

import (
	"fmt"
	"strconv"

	"github.com/jackc/pgx"
)

func WriteBookToDb(pool *pgx.Conn, book Book) (Book, error) {
	bookIdStr := strconv.FormatInt(book.BookId, 10)

	err := pool.QueryRow(fmt.Sprintf("INSERT INTO books (book_id, title, author) VALUES ('%s','%s', '%s') RETURNING book_id;", bookIdStr, book.Title, book.Author)).Scan(&book.BookId)

	if err != nil {
		return Book{}, err
	}

	return book, nil
}

func GetBookFromDB(pool *pgx.Conn, book_id int64) (Book, error) {
	book := Book{
		BookId: book_id,
	}

	var bid int
	err := pool.QueryRow(fmt.Sprintf("SELECT book_id, title, author FROM books where book_id = '%d';", book_id)).Scan(&bid, &book.Title, &book.Author)

	if err != nil {
		panic(err)
	}

	return book, nil
}

func GetAllBooksFromDB(pool *pgx.Conn) ([]Book, error) {
	rows, err := pool.Query("SELECT book_id, title, author FROM books;")

	if err != nil {
		panic(err)
	}

	results := []Book{}

	defer rows.Close()

	for rows.Next() {
		book := Book{}
		err := rows.Scan(&book.BookId, &book.Title, &book.Author)

		if err != nil {
			panic(err)
		}

		results = append(results, book)
	}

	return results, nil
}
