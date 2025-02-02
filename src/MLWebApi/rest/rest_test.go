package rest_test

import (
	"bytes"
	"encoding/json"
	"mlwebapi/rest"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func setupRouter() *gin.Engine {
	gin.SetMode(gin.TestMode)
	r := gin.Default()
	r.GET("/photo/:token/:id", rest.GetPhoto)
	r.GET("/photos/:token", rest.GetUserPhotos)
	r.GET("/photos/unprocessed/:token", rest.GetUserUnprocessedPhotos)
	r.POST("/photo/:token", rest.PostPhoto)
	r.DELETE("/photo/:token/:id", rest.DeletePhoto)
	r.PUT("/model", rest.PutModel)
	r.GET("/photo/order/:token/:id", rest.GetPhotoOrder)
	r.PUT("/photo/order", rest.PutOrder)
	return r
}

func TestGetPhoto(t *testing.T) {
	r := setupRouter()
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/photo/test-token/test-id", nil)
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusForbidden, w.Code)
}

func TestPostPhoto(t *testing.T) {
	r := setupRouter()
	w := httptest.NewRecorder()

	photoReq := map[string][]byte{"photo": {0x12, 0x34, 0x56}}
	jsonValue, _ := json.Marshal(photoReq)

	req, _ := http.NewRequest("POST", "/photo/test-token", bytes.NewBuffer(jsonValue))
	req.Header.Set("Content-Type", "application/json")

	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusAccepted, w.Code)
}

func TestDeletePhoto(t *testing.T) {
	r := setupRouter()
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("DELETE", "/photo/test-token/test-id", nil)
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusForbidden, w.Code)
}

func TestPutModel(t *testing.T) {
	r := setupRouter()
	w := httptest.NewRecorder()

	modelReq := map[string]interface{}{
		"id":    "test-id",
		"token": "test-token",
		"model": []byte{0x12, 0x34, 0x56},
	}
	jsonValue, _ := json.Marshal(modelReq)

	req, _ := http.NewRequest("PUT", "/model", bytes.NewBuffer(jsonValue))
	req.Header.Set("Content-Type", "application/json")

	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusForbidden, w.Code)
}

func TestGetUserPhotos(t *testing.T) {
	r := setupRouter()
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/photos/test-token", nil)
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
}

func TestGetPhotoOrder(t *testing.T) {
	r := setupRouter()
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/photo/order/test-token/test-id", nil)
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusForbidden, w.Code)
}

func TestPutOrder(t *testing.T) {
	r := setupRouter()
	w := httptest.NewRecorder()

	orderReq := map[string]interface{}{
		"id":    "test-id",
		"token": "test-token",
		"order": 1,
	}
	jsonValue, _ := json.Marshal(orderReq)

	req, _ := http.NewRequest("PUT", "/photo/order", bytes.NewBuffer(jsonValue))
	req.Header.Set("Content-Type", "application/json")

	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusForbidden, w.Code)
}
