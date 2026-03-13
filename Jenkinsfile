pipeline {
    agent any

    environment {
        CARGO_HOME = "${WORKSPACE}/.cargo"
        RUSTUP_HOME = "${WORKSPACE}/.rustup"
    }

    stages {
        stage('Setup') {
            steps {
                sh 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable'
                sh 'export PATH="${CARGO_HOME}/bin:${PATH}" && rustup component add clippy rustfmt'
            }
        }

        stage('Check') {
            steps {
                sh 'export PATH="${CARGO_HOME}/bin:${PATH}" && cargo check --workspace'
            }
        }

        stage('Format') {
            steps {
                sh 'export PATH="${CARGO_HOME}/bin:${PATH}" && cargo fmt --all -- --check'
            }
        }

        stage('Lint') {
            steps {
                sh 'export PATH="${CARGO_HOME}/bin:${PATH}" && cargo clippy --workspace -- -D warnings'
            }
        }

        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh 'export PATH="${CARGO_HOME}/bin:${PATH}" && cargo test --workspace'
                    }
                }
                stage('Doc Tests') {
                    steps {
                        sh 'export PATH="${CARGO_HOME}/bin:${PATH}" && cargo test --workspace --doc'
                    }
                }
            }
        }

        stage('Documentation') {
            steps {
                sh 'export PATH="${CARGO_HOME}/bin:${PATH}" && cargo doc --workspace --no-deps'
            }
        }

        stage('Benchmark') {
            when {
                branch 'main'
            }
            steps {
                sh 'export PATH="${CARGO_HOME}/bin:${PATH}" && cargo bench --workspace 2>&1 | tee benchmark-results.txt'
                archiveArtifacts artifacts: 'benchmark-results.txt', fingerprint: true
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        success {
            echo 'Build succeeded!'
        }
        failure {
            echo 'Build failed!'
        }
    }
}
