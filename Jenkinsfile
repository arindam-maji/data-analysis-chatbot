pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Setup Environment') {
            steps {
                bat 'py -m pip install --upgrade pip'
                bat 'py -m pip install -r requirements.txt'
            }
        }
        
        stage('Test') {
            steps {
                bat 'echo Tests would go here'
                // Future: Add pytest
                // bat 'pytest'
            }
        }
        
        stage('Build Docker Image') {
            steps {
                bat 'docker-compose build'
            }
        }

        stage('Deploy with Docker Compose') {
            steps {
                bat 'docker-compose down || exit 0'  // '|| exit 0' so it doesn’t fail if nothing to stop
                bat 'docker-compose up -d'
            }
        }
    }

    post {
        success {
            echo 'Pipeline executed successfully!'
        }
        failure {
            echo 'Pipeline execution failed!'
        }
    }
}
