package datamodel

import (
	"context"
	"fmt"
	"os"

	"github.com/jackc/pgx/v5"
)

// infrastructure
var conn *pgx.Conn
var err error

// checkError checks if there error and quits the application if yes
func CheckError(err error) {
	if err != nil && err.Error() != "no rows in result set" {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func Connect() {
	conn, err = pgx.Connect(context.Background(), os.Getenv("DATABASE_URL"))
	CheckError(err)
}

// Close closes the database connection
func Close() error {
	return conn.Close(context.Background())
}

func SelectPhoto(id string, token string) pgx.Row {
	return conn.QueryRow(context.Background(), "select photo from photo3d where id = $1 and token = $2;", id, token)
}
func SelectPhotos(token string) pgx.Rows {
	rows, err := conn.Query(context.Background(), "select id, photo, model from photo3d where token=$1 and not model is null;", token)
	CheckError(err)
	return rows
}

func SelectUserUnprocessedPhotos(token string) pgx.Rows {
	rows, err := conn.Query(context.Background(), "select id, photo, order_num from photo3d where token=$1 and model is null;", token)
	CheckError(err)
	return rows
}

func InsertPhoto(token string, photo []byte) pgx.Row {
	return conn.QueryRow(context.Background(), "insert into photo3d (token, photo) VALUES ($1, $2) RETURNING id, order_num;", token, photo)
}

func DeletePhoto(id string, token string) pgx.Row {
	return conn.QueryRow(context.Background(), "delete from photo3d where id = $1 and token = $2;", id, token)
}

func InsertModel(model []byte, id string) pgx.Row {
	return conn.QueryRow(context.Background(), "update photo3d set model = $1 where id = $2;", model, id)
}

func SelectPhotoOrder(id string, token string) pgx.Row {
	return conn.QueryRow(context.Background(), "select order_num from photo3d where id = $1 and token = $2;", id, token)
}

func IncreaseOrderOver(target int) string {
	var text string
	row := conn.QueryRow(context.Background(), "update photo3d set order_num = order_num + 1 where order_num >= $1;", target)
	err := row.Scan(&text)
	CheckError(err)
	return text
}

func SetOrder(target int, id string) pgx.Row {
	return conn.QueryRow(context.Background(), "update photo3d set order_num  = $1 where id = $2;", target, id)
}

// SelectFirstUnprocessedPhotoOrder returns last unprocessed(where model is null) photo order and returns its int number
func SelectFirstUnprocessedPhotoOrder() int {
	var order int
	err = conn.QueryRow(context.Background(), "select min(order_num) from photo3d where model is null;").Scan(&order)
	if err != nil {
		return 0
	}
	return order
}

// SelectNUnprocessedPhotos returns last unprocessed(where model is null) photo order and returns its int number
func SelectNUnprocessedPhotos(n int) pgx.Rows {
	var rows pgx.Rows
	rows, err = conn.Query(context.Background(), "select id, photo from photo3d where model is null order by order_num asc limit $1;", n)
	CheckError(err)
	return rows
}
