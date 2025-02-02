package datamodel

import (
	"context"
	"fmt"
	"os"

	"github.com/jackc/pgx/v5"
)

var conn *pgx.Conn

// CheckError checks if there is an error and quits the application if needed
func CheckError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// Connect initializes the database connection pool
func Connect() {
	var err error
	conn, err = pgx.Connect(context.Background(), os.Getenv("DATABASE_URL"))
	CheckError(err)
}

// Close closes the database connection pool
func Close() error {
	if conn != nil {
		conn.Close(context.Background())
	}
	return nil
}

// GetDB returns the database pool instance
func GetDB() *pgx.Conn {
	return conn
}

// SetDB allows setting a mock database pool (for testing)
func SetDB(mockConn *pgx.Conn) {
	conn = mockConn
}

// SelectPhoto retrieves a single photo
func SelectPhoto(id string, token string) pgx.Row {
	return conn.QueryRow(context.Background(), "SELECT photo FROM photo3d WHERE id = $1 AND token = $2;", id, token)
}

// SelectPhotos retrieves processed photos
func SelectPhotos(token string) (pgx.Rows, error) {
	return conn.Query(context.Background(), "SELECT id, photo, model FROM photo3d WHERE token=$1 AND model IS NOT NULL;", token)
}

// SelectUserUnprocessedPhotos retrieves unprocessed photos
func SelectUserUnprocessedPhotos(token string) (pgx.Rows, error) {
	return conn.Query(context.Background(), "SELECT id, photo, order_num FROM photo3d WHERE token=$1 AND model IS NULL;", token)
}

// InsertPhoto inserts a new photo and returns its ID and order number
func InsertPhoto(token string, photo []byte) pgx.Row {
	return conn.QueryRow(context.Background(), "INSERT INTO photo3d (token, photo) VALUES ($1, $2) RETURNING id, order_num;", token, photo)
}

// DeletePhoto removes a photo
func DeletePhoto(id string, token string) pgx.Row {
	return conn.QueryRow(context.Background(), "DELETE FROM photo3d WHERE id = $1 AND token = $2 RETURNING 'deleted';", id, token)
}

// InsertModel updates a model for a photo
func InsertModel(model []byte, id string) pgx.Row {
	return conn.QueryRow(context.Background(), "UPDATE photo3d SET model = $1 WHERE id = $2 RETURNING 'updated';", model, id)
}

// SelectPhotoOrder retrieves the order of a photo
func SelectPhotoOrder(id string, token string) pgx.Row {
	return conn.QueryRow(context.Background(), "SELECT order_num FROM photo3d WHERE id = $1 AND token = $2;", id, token)
}

// IncreaseOrderOver increases order_num for all photos above a target
func IncreaseOrderOver(target int) (string, error) {
	row := conn.QueryRow(context.Background(), "UPDATE photo3d SET order_num = order_num + 1 WHERE order_num >= $1 RETURNING 'updated';", target)
	var result string
	err := row.Scan(&result)
	return result, err
}

// SetOrder updates the order number of a photo
func SetOrder(target int, id string) pgx.Row {
	return conn.QueryRow(context.Background(), "UPDATE photo3d SET order_num = $1 WHERE id = $2 RETURNING 'updated';", target, id)
}

// SelectFirstUnprocessedPhotoOrder returns the first unprocessed photo order
func SelectFirstUnprocessedPhotoOrder() (int, error) {
	var order int
	err := conn.QueryRow(context.Background(), "SELECT MIN(order_num) FROM photo3d WHERE model IS NULL;").Scan(&order)
	return order, err
}
