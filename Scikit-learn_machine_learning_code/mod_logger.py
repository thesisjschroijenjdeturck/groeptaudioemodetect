# This is the logger module
logFilename = "./EmoDetectAutoML.log"

def logMessage( tag, message ):
    with open( logFilename, "a" ) as f:
        f.write( "MESSAGE: %s: %s\n" % ( tag, message ) )

def logDebug( tag, message ):
    with open( logFilename,"a" ) as f:
        f.write( "DEBUG: %s: %s\n" % ( tag, message ) )

def logError( tag, message ):
    with open( logFilename,"a" ) as f:
        f.write( "ERROR: %s: %s\n" % ( tag, message ) )