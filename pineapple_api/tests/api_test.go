package tests

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"testing"

	c "pineapple/backend/src/controller"
	"pineapple/backend/src/model"

	"github.com/huandu/go-assert"
	"github.com/jackc/pgx"
)

func TestAddBook(t *testing.T) {
	assert := assert.New(t)

	cfg := pgx.ConnConfig{
		User:     "postgres",
		Database: "backendbootcamp",
		Password: "password",
		Host:     "127.0.0.1",
		Port:     5432,
	}

	conn, err := pgx.Connect(cfg)

	if err != nil {
		fmt.Fprintf(os.Stderr, "Unable to connect to database: %v\n", err)
		os.Exit(1)
	}

	defer conn.Close()

	m := &model.PgModel{
		Conn: conn,
	}
	c := &c.PgController{
		Model: m,
	}

	router := c.Serve()

	w := httptest.NewRecorder()

	req := httptest.NewRequest("POST", "/v1/addBook", bytes.NewBuffer([]byte(fmt.Sprintf(`{"id":%v,"title":"%s","author":"%s"}`, 1738, "The Lightning Thief", "Rick Riordan"))))

	req.Header.Set("Content-Type", "application/json")

	router.ServeHTTP(w, req)

	assert.Equal(200, w.Code)

	var book model.Book

	if e := json.Unmarshal(w.Body.Bytes(), &book); e != nil {
		panic(err)
	}

	assert.Equal(model.Book{
		BookId: 1738,
		Title:  "The Lightning Thief",
		Author: "Rick Riordan",
	}, book)
}
