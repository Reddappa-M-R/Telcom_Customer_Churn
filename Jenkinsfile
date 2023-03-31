pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Train model') {
            steps {
                sh 'python3 train.py'
            }
        }

        stage('Evaluate model') {
            steps {
                sh 'python3 evaluate.py'
            }
        }

        stage('Docker image') {
            steps {
                sh 'docker build -t telcom_customer_churn:latest .'
            }
        }

        stage('Docker container and deploy') {
            steps {
                sh 'docker run -p 5000:5000 --name telcom_customer_churn telcom_customer_churn:latest'
            }
        }
    }
}
