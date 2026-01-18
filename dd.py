from enhanced_sms_detector import EnhancedSMSFraudDetector

detector = EnhancedSMSFraudDetector()
detector.load_model('enhanced_sms_detector.pkl')
result = detector.predict_sms("Your message here")
print(result['prediction']) 