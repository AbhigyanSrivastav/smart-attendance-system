import { useState, useRef } from 'react';
import * as XLSX from 'xlsx';
import {
  Box,
  Button,
  Typography,
  Card,
  CardMedia,
  IconButton,
} from '@mui/material';
import { Upload as UploadIcon, Download as DownloadIcon } from '@mui/icons-material';

const ImageUploadAndRecognition = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [recognizedNames, setRecognizedNames] = useState([]);
  const fileInputRef = useRef(null);
//will be changed everytime the backend runs
  const BACKEND_URL = 'https://a159-35-247-182-201.ngrok-free.app/recognize';

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => setSelectedImage(e.target.result);
    reader.readAsDataURL(file);
  };

  const handleSubmitImage = async () => {
    if (!selectedImage) return;

    try {
      const formData = new FormData();
      const blob = await fetch(selectedImage).then(r => r.blob());
      formData.append('image', blob, 'uploaded_image.jpg');

      const response = await fetch(BACKEND_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error ${response.status}`);
      }

      const data = await response.json();
      if (data.status === 'success') {
        setRecognizedNames(data.recognized_names);
      } else {
        console.error("Backend returned an error:", data.error);
      }
    } catch (error) {
      console.error('Error recognizing image:', error);
    }
  };

  const handleDownloadExcel = () => {
    const dataForExcel = [
      ['Name', 'Attendance'],
      ...recognizedNames.map((name) => [name, 'Present']),
    ];

    const sheet = XLSX.utils.aoa_to_sheet(dataForExcel);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, sheet, 'Attendance');
    XLSX.writeFile(workbook, 'attendance.xlsx');
  };

  return (
    <Box sx={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: 'background.default', p: 3 }}>
      <Card sx={{ maxWidth: 400, p: 3, textAlign: 'center' }}>
        <Typography variant="h6" gutterBottom>
          Upload and Recognize Image
        </Typography>

        <Box sx={{ border: '2px dashed grey', borderRadius: 2, p: 2, mb: 3, cursor: 'pointer', textAlign: 'center' }} onClick={() => fileInputRef.current.click()}>
          {selectedImage ? (
            <CardMedia component="img" src={selectedImage} alt="Uploaded" sx={{ maxHeight: 200, objectFit: 'contain' }} />
          ) : (
            <IconButton color="primary" aria-label="upload picture" component="span">
              <UploadIcon sx={{ fontSize: 50 }} />
            </IconButton>
          )}
          <input type="file" accept="image/*" ref={fileInputRef} onChange={handleImageUpload} style={{ display: 'none' }} />
        </Box>

        <Button variant="contained" color="primary" fullWidth disabled={!selectedImage} onClick={handleSubmitImage} sx={{ mb: 2 }}>
          {selectedImage ? 'Submit' : 'Upload'}
        </Button>

        {recognizedNames.length > 0 && (
          <>
            <Typography variant="subtitle1" gutterBottom>
              Recognized Names:
            </Typography>
            {recognizedNames.map((name, idx) => (
              <Typography key={idx} variant="body2">
                {name}
              </Typography>
            ))}
            <Button variant="outlined" color="secondary" fullWidth startIcon={<DownloadIcon />} onClick={handleDownloadExcel} sx={{ mt: 2 }}>
              Download Excel
            </Button>
          </>
        )}
      </Card>
    </Box>
  );
};

export default ImageUploadAndRecognition;