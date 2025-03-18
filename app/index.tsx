import React, { useState, useEffect } from 'react';
import { View, Button, Text, Image, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

export default function App() {
  const [image, setImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);

  // Request permissions for camera and gallery
  const requestPermissions = async () => {
    const { status: cameraStatus } = await ImagePicker.requestCameraPermissionsAsync();
    const { status: mediaLibraryStatus } = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (cameraStatus !== 'granted' || mediaLibraryStatus !== 'granted') {
      Alert.alert('Permission to access camera or media library is required!');
    }
  };

  // Function to take a photo using the camera
  const takePhoto = async () => {
    let result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    console.log("Camera result:", result); // Debug log to check the result

    // Check if the photo was not canceled and assets exist
    if (!result.canceled && result.assets && result.assets.length > 0) {
      const uri = result.assets[0].uri; // Get URI from the first asset
      setImage(uri);
      uploadImage(uri); // Upload the image immediately
    } else {
      Alert.alert('Photo capture cancelled!');
    }
  };

  // Send the image to the Flask server for prediction
  const uploadImage = async (uri: string) => {
    if (!uri) {
      Alert.alert("Please take a photo first!");
      return;
    }

    let localUri = uri;
    let filename = localUri.split("/").pop();
    let type = "image/jpeg"; // Adjust based on the actual image format if necessary

    const formData = new FormData();
    formData.append("face", { uri: localUri, name: filename, type });

    try {
      const response = await fetch("http://YOUR_IP_HERE:5000/upload", {

        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Server error");

      const responseJson = await response.json();
      setPrediction(responseJson.results.face);
      Alert.alert("Prediction: " + responseJson.results.face);
    } catch (error) {
      console.error("Upload Error:", error);
      Alert.alert("Upload Error: " + error.message);
    }
  };

  // Request permissions on component mount
  useEffect(() => {
    requestPermissions();
  }, []);

  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>Take a Photo for Prediction:</Text>
      <Button title="Take a Photo" onPress={takePhoto} />

      {image && (
        <View style={{ marginTop: 20 }}>
          <Text>Captured Image:</Text>
          <Image source={{ uri: image }} style={{ width: 200, height: 200 }} />
        </View>
      )}

      {prediction && (
        <View style={{ marginTop: 20 }}>
          <Text>Prediction Result:</Text>
          <Text>{prediction}</Text>
        </View>
      )}
    </View>
  );
}
