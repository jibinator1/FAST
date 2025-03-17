import React, { useState } from 'react';
import { View, Button, Text, TextInput, Switch, ScrollView, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Picker } from '@react-native-picker/picker';

export default function App() {
  const [image, setImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    age: '',
    hypertension: false,
    heart_disease: false,
    avg_glucose_level: '',
    bmi: '',
    gender: 'Male',
    ever_married: 'No',
    work_type: 'Private',
    residence_type: 'Urban', // Fixed key name
    smoking_status: 'never smoked'
  });

  const handleInputChange = (name: string, value: string | boolean) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert("Permission required", "Camera access is needed to take photos.");
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
    }
  };

  const uploadData = async () => {
    const payload = {
      ...formData,
      hypertension: formData.hypertension ? "true" : "false",
      heart_disease: formData.heart_disease ? "true" : "false",
      image: image
        ? {
            uri: image,
            name: image.split('/').pop(),
            type: 'image/jpeg'
          }
        : null
    };

    try {
      const form = new FormData();
      Object.entries(payload).forEach(([key, value]) => {
        if (value !== null) {
          form.append(key, value);
        }
      });

      const response = await fetch("http://ADDYOURIPHERE/upload", {
        method: "POST",
        body: form,
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      if (!response.ok) {
        throw new Error(`Server Error: ${response.status}`);
      }

      const result = await response.json();
      setPrediction(result.prediction);
      Alert.alert(`Stroke Risk: ${result.risk_level} (${result.probability}%)`);
    } catch (error: any) {
      Alert.alert("Error", error.message || "Something went wrong.");
    }
  };

  return (
    <ScrollView contentContainerStyle={{ padding: 20 }}>
      <Text>Age:</Text>
      <TextInput
        placeholder="Age"
        keyboardType="numeric"
        onChangeText={t => handleInputChange('age', t)}
      />

      <View style={{ flexDirection: 'row', alignItems: 'center' }}>
        <Text>Hypertension:</Text>
        <Switch
          value={formData.hypertension}
          onValueChange={v => handleInputChange('hypertension', v)}
        />
      </View>

      <View style={{ flexDirection: 'row', alignItems: 'center' }}>
        <Text>Heart Disease:</Text>
        <Switch
          value={formData.heart_disease}
          onValueChange={v => handleInputChange('heart_disease', v)}
        />
      </View>

      <Text>Avg Glucose Level:</Text>
      <TextInput
        placeholder="Glucose Level"
        keyboardType="numeric"
        onChangeText={t => handleInputChange('avg_glucose_level', t)}
      />

      <Text>BMI:</Text>
      <TextInput
        placeholder="BMI"
        keyboardType="numeric"
        onChangeText={t => handleInputChange('bmi', t)}
      />

      <Text>Gender:</Text>
      <Picker
        selectedValue={formData.gender}
        onValueChange={t => handleInputChange('gender', t)}
      >
        <Picker.Item label="Male" value="Male" />
        <Picker.Item label="Female" value="Female" />
        <Picker.Item label="Other" value="Other" />
      </Picker>

      <Text>Ever Married:</Text>
      <Picker
        selectedValue={formData.ever_married}
        onValueChange={t => handleInputChange('ever_married', t)}
      >
        <Picker.Item label="No" value="No" />
        <Picker.Item label="Yes" value="Yes" />
      </Picker>

      <Text>Work Type:</Text>
      <Picker
        selectedValue={formData.work_type}
        onValueChange={t => handleInputChange('work_type', t)}
      >
        <Picker.Item label="Private" value="Private" />
        <Picker.Item label="Self-employed" value="Self-employed" />
        <Picker.Item label="Government Job" value="Govt_job" />
        <Picker.Item label="Children" value="Children" />
        <Picker.Item label="Never Worked" value="Never_worked" />
      </Picker>

      <Text>Residence Type:</Text>
      <Picker
        selectedValue={formData.residence_type}
        onValueChange={t => handleInputChange('residence_type', t)}
      >
        <Picker.Item label="Urban" value="Urban" />
        <Picker.Item label="Rural" value="Rural" />
      </Picker>

      <Text>Smoking Status:</Text>
      <Picker
        selectedValue={formData.smoking_status}
        onValueChange={t => handleInputChange('smoking_status', t)}
      >
        <Picker.Item label="Never Smoked" value="never smoked" />
        <Picker.Item label="Formerly Smoked" value="formerly smoked" />
        <Picker.Item label="Smokes" value="smokes" />
      </Picker>

      <Button title="Take Photo" onPress={takePhoto} />
      {image && <Text>Photo Selected</Text>}

      <Button title="Submit" onPress={uploadData} />
    </ScrollView>
  );
}
