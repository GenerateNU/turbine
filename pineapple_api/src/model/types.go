package model

type Book struct {
	BookId int64  `json:"id" db:"book_id"`
	Title  string `json:"title" db:"title"`
	Author string `json:"author" db:"author"`
}

type Student struct {
	NUID int64  `json:"nuid" db:"nuid"`
	Name string `json:"name" db:"name"`
}
