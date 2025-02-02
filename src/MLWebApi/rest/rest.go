package rest

import (
	"bufio"

	"fmt"
	"net/http"
	"os"
	"slices"

	"github.com/runderagon-dev/mlwebapi/datamodel"

	"github.com/gin-gonic/gin"
)

type Photo3D struct {
	ID    string `json:"id"`
	Photo []byte `json:"photo"`
	Model []byte `json:"model"`
}

type Photo3D_Unprocessed struct {
	ID    string `json:"id"`
	Photo []byte `json:"photo"`
	Order int    `json:"order"`
}

type PhotoRequest struct {
	Photo []byte `json:"photo"`
}
type Photo struct {
	ID    string `json:"id"`
	Photo []byte `json:"photo"`
}

type Model struct {
	ID    string `json:"id"`
	Token string `json:"token"`
	Model []byte `json:"model"`
}

type PhotoOrder struct {
	ID    string `json:"id"`
	Order int    `json:"order"`
}
type PhotoChangeOrder struct {
	ID    string `json:"id"`
	Token string `json:"token"`
	Order int    `json:"order"`
}
type PhotoId struct {
	ID string `json:"id"`
}

type ErrorMsg struct {
	Error string `json:"error"`
}
type StatusMsg struct {
	Status string `json:"status"`
}

var models []string

// checkError checks if there error which means tokens are not equal, but not quits application and sends Forbidden status error as an answer
func checkAccess(err error, c *gin.Context, returnOnSucces bool, token string) {
	if err != nil || (token != "" && slices.Contains(models, token)) {
		msg := ErrorMsg{
			Error: "Unauthorized access",
		}
		c.IndentedJSON(http.StatusForbidden, msg)
	} else {
		if returnOnSucces {
			status := StatusMsg{Status: "success"}
			c.IndentedJSON(http.StatusOK, status)
		}
	}
}

func Load() {
	//os.Setenv("DATABASE_URL", "postgres://postgres:denis471248@localhost:5432/photos_3d")
	datamodel.Connect()
	file, err := os.Open("ModelsTokens.txt")
	datamodel.CheckError(err)
	if err != nil {
		fmt.Println("Ошибка при открытии файла:", err)
		return
	}
	defer file.Close()

	// Создаем новый сканер
	scanner := bufio.NewScanner(file)

	// Читаем файл построчно
	for scanner.Scan() {
		line := scanner.Text() // Получаем строку
		fmt.Println("token :", line)
		models = append(models, line)
	}
}

func UnLoad() {
	datamodel.Close()
}

// GetPhoto responds with id and photo as byte array of photo with given id and token
func GetPhoto(c *gin.Context) {
	token := c.Param("token")
	id := c.Param("id")

	var photo []byte

	err := datamodel.SelectPhoto(id, token).Scan(&photo)

	checkAccess(err, c, false, token)

	photoJSON := Photo{
		ID:    id,
		Photo: photo,
	}
	c.IndentedJSON(http.StatusOK, photoJSON)
}

// GetUserPhotos responds with the list of all photos with models as JSON by the token param.
func GetUserPhotos(c *gin.Context) {
	token := c.Param("token")

	rows := datamodel.SelectPhotos(token)

	var photos []Photo3D

	for rows.Next() {
		var photo Photo3D
		err := rows.Scan(&photo.ID, &photo.Photo, &photo.Model)
		datamodel.CheckError(err)
		photos = append(photos, photo)
	}
	c.IndentedJSON(http.StatusOK, photos)
}

// GetUserUnprocessedPhotos responds with the list of all photos that still unprocessed as JSON by the token param.
func GetUserUnprocessedPhotos(c *gin.Context) {
	token := c.Param("token")

	rows := datamodel.SelectUserUnprocessedPhotos(token)

	var photos []Photo3D_Unprocessed

	for rows.Next() {
		var photo Photo3D_Unprocessed
		err := rows.Scan(&photo.ID, &photo.Photo, &photo.Order)
		datamodel.CheckError(err)
		photos = append(photos, photo)
	}
	c.IndentedJSON(http.StatusOK, photos)
}

// PostPhoto loads photo and returs its id and order
func PostPhoto(c *gin.Context) {
	token := c.Param("token")

	var photo PhotoRequest

	err := c.BindJSON(&photo)
	//datamodel.CheckError(err)

	var order int
	var id string

	err = datamodel.InsertPhoto(token, photo.Photo).Scan(&id, &order)
	datamodel.CheckError(err)

	photoOrder := PhotoOrder{
		ID:    id,
		Order: order,
	}
	c.IndentedJSON(http.StatusAccepted, photoOrder)
}

// DeletePhoto deletes photo by given id if token of user is euqal to the one who loaded it and returns success, otherwise it returns forbidden status with error
func DeletePhoto(c *gin.Context) {
	token := c.Param("token")
	id := c.Param("id")

	var text string

	err := datamodel.DeletePhoto(id, token).Scan(&text)
	checkAccess(err, c, true, token)
}

// PutModel responds with id and photo as byte array of photo with given id and token
func PutModel(c *gin.Context) {
	var text string
	var model Model

	err := c.BindJSON(&model)
	//datamodel.CheckError(err)

	err = datamodel.InsertModel(model.Model, model.ID).Scan(&text)
	checkAccess(err, c, true, model.Token)
}

// GetPhotoOrder responds with id and order(to process, 0 means photo is processed right now) of photo with given id and token
func GetPhotoOrder(c *gin.Context) {
	token := c.Param("token")
	id := c.Param("id")

	var order int

	err := datamodel.SelectPhotoOrder(id, token).Scan(&order)

	checkAccess(err, c, false, token)

	first_order := datamodel.SelectFirstUnprocessedPhotoOrder()

	photoOrder := PhotoOrder{
		ID:    id,
		Order: max(0, order-first_order-len(models)),
	}

	c.IndentedJSON(http.StatusOK, photoOrder)
}

// PutOrder sets the order to photo, possible values (1,int_max) 1 means it will be processed after any other photo will be processed
func PutOrder(c *gin.Context) {
	var text string
	var photo PhotoChangeOrder

	err := c.BindJSON(&photo)
	//datamodel.CheckError(err)
	target := datamodel.SelectFirstUnprocessedPhotoOrder() + len(models) + photo.Order
	datamodel.IncreaseOrderOver(target)
	err = datamodel.SetOrder(target, photo.ID).Scan(&text)
	checkAccess(err, c, true, photo.Token)
}
