package controller

import (
	"fmt"
	"net/http"
	"strconv"

	"pineapple/backend/src/model"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

type Controller interface {
	Serve() *gin.Engine
}

type PgController struct {
	model.Model
}

func (pg *PgController) Serve() *gin.Engine {
	r := gin.Default()

	r.Use(cors.Default())
	r.GET("/v1/books/:bookId", func(c *gin.Context) {
		id := c.Param("bookId")
		intId, err := strconv.Atoi(id)

		if err != nil {
			panic(err)
		}
		c.JSON(http.StatusOK, pg.Book(int64(intId)))
	})
	r.GET("/v1/books/", func(c *gin.Context) {
		books, err := pg.AllBooks()
		if err != nil {
			c.JSON(http.StatusInternalServerError, "Oops")
		}
		c.JSON(http.StatusOK, books)
	})

	r.POST("/v1/addBook", func(c *gin.Context) {
		var book model.Book

		if err := c.BindJSON(&book); err != nil {
			c.JSON(http.StatusBadRequest, "Failed to unmarshal book")
			fmt.Print(err)
			return
		}
		insertedBook, err := pg.AddBook(book)

		if err != nil {
			c.JSON(http.StatusBadRequest, "Failed to add a book")
			panic(err)
		}

		c.JSON(http.StatusOK, insertedBook)
	})

	return r
}
