FROM maven:3.9.4-eclipse-temurin-21 AS build
WORKDIR /app
COPY pom.xml .
COPY src ./src
RUN mvn clean package -DskipTests

FROM eclipse-temurin:21-jre
WORKDIR /app
COPY --from=build /app/target/FedChain-1.0-SNAPSHOT.jar ./fedchain.jar
EXPOSE 9000
CMD ["java", "-cp", "fedchain.jar", "network.BootstrapServer"]
