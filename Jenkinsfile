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
                sh 'docker build -t telco_customer_churn:latest .'
            }
        }

        stage('Docker container and deploy') {
            steps {
                sh 'docker run -p 8502:8502 --name telco_customer_churn telco_customer_churn:latest'
            }
        }
    }
}
