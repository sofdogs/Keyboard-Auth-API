require("dotenv").config() // load .env variables
const mongoose = require("mongoose") //import fresh mongoose object
const {log} = require("mercedlogger") // import merced logger

//DESTRUCTURE ENV VARIABLES
const {DATABASE_URL} = process.env 

// CONNECT TO MONGO
mongoose.connect = mongoose.connect(DATABASE_URL, {useNewUrlParser: true, useUnifiedTopology: true})

// CONNECTION EVENTS
mongoose.connection
.on("open", () => log.green("DATABASE STATE", "Connection Open"))
.on("close", () => log.magenta("DATABASE STATE", "Connection Open"))
.on("error", (error) => log.red("DATABASE STATE", error))

// EXPORT CONNECTION
module.exports = mongoose

/*
const Mongoose = require("mongoose")

const localDB = "mongodb+srv://csproject:keystroke@cluster0.tml1aoh.mongodb.net/?retryWrites=true&w=majority"
const connectDB = async () => {
  await Mongoose.connect(localDB, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  });
  console.log("MongoDB Connected")
}
module.exports = connectDB
*/