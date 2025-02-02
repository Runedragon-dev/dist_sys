package main

import (
	"fmt"
	"mlwebapi/rest"
	"os"

	"github.com/gin-gonic/gin"
)

func main() {
	fmt.Println("running")
	os.Setenv("DATABASE_URL", "postgres://postgres:denis471248@localhost:5432/photos_3d")
	router := gin.Default()
	rest.Load()
	defer rest.UnLoad()

	router.GET("/photo/:id/:token/", rest.GetPhoto)
	router.GET("/photos/:token", rest.GetUserPhotos)
	router.GET("/unprocessed/:token", rest.GetUserUnprocessedPhotos)
	router.POST("/postphoto/:token", rest.PostPhoto)
	router.DELETE("/deletephoto/:id/:token", rest.DeletePhoto)
	router.PUT("/putmodel", rest.PutModel)
	router.GET("/getorder/:id/:token", rest.GetPhotoOrder)
	router.PUT("/putorder/:id/:token", rest.PutOrder)
	router.Run("localhost:8080")
}
